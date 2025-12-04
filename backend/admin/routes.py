from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import text, create_engine, or_, and_ 
from sqlalchemy.pool import NullPool
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime, timedelta

from auth.dependencies import get_current_admin
from auth.models import Admin, User, UserRegistrationRequest, Credentials
from db import get_auth_db

router = APIRouter(prefix="/admin", tags=["admin"])

# Pydantic models for admin operations
class UserUpdate(BaseModel):
    provincial_access: Optional[str]
    municipal_access: Optional[str]

class UserResponse(BaseModel):
    id: int
    user_name: str
    provincial_access: Optional[str]
    municipal_access: Optional[str]

class AdminInfo(BaseModel):
    id: int
    first_name: str
    last_name: str
    email: str
    contact_number: str
    user_name: str

class RegistrationReviewRequest(BaseModel):
    request_id: int
    action: str  # 'approve' or 'reject'
    remarks: Optional[str] = None
    provincial_access: Optional[str] = None
    municipal_access: Optional[str] = None

def strip_suffix(code: str | None) -> str | None:    #nirereturn lang dto ung PSA Code lang example, PH0403406_% (ang ireread lang nya us ung PSA Code)
    if not code:
        return code
    if code.strip().lower() == "all":
        return "ALL"
    return code.split("_")[0]

# Admin dashboard endpoint
@router.get("/dashboard")
async def admin_dashboard(
    current_admin: Admin = Depends(get_current_admin)
):
    """Get admin dashboard information"""
    return {
        "message": f"Welcome Admin {current_admin.first_name} {current_admin.last_name}",
        "admin_info": {
            "id": current_admin.id,
            "full_name": f"{current_admin.first_name} {current_admin.last_name}",
            "email": current_admin.email,
            "contact": current_admin.contact_number,
            "username": current_admin.user_name
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/users")
async def get_all_users(
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_auth_db)
):
    users = db.query(User).all()
    return [
        {
            "id": u.id,
            "user_name": u.user_name,
            "first_name": u.first_name,
            "last_name": u.last_name,
            "full_name": f"{u.first_name or ''} {u.last_name or ''}".strip() or None,
            "email": u.email,
            "contact_number": u.contact_number,
            "provincial_access": strip_suffix(u.provincial_access),
            "municipal_access": strip_suffix(u.municipal_access)
        }
        for u in users
    ]

@router.get("/users/pending")
async def get_pending_users(current_admin: Admin = Depends(get_current_admin), db: Session = Depends(get_auth_db)):
    users = db.query(User).filter(
        (User.provincial_access.is_(None)) |
        (User.municipal_access.is_(None))
    ).all()
    return {
        "pending_users": [
            {
                "id": u.id,
                "username": u.user_name,
                "provincial_access": strip_suffix(u.provincial_access),
                "municipal_access": strip_suffix(u.municipal_access),
                "status": "No access" if u.provincial_access is None else "Partial access"
            }
            for u in users
        ],
        "total_pending": len(users)
    }

@router.put("/users/{user_id}/access")
async def update_user_access(
    user_id: int,
    user_update: UserUpdate,
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_auth_db)
):
    """Update user's provincial and municipal access"""
    user = db.query(User).filter(User.id == user_id).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # ✅ Always strip suffix before saving
    if user_update.provincial_access is not None:
        user.provincial_access = strip_suffix(user_update.provincial_access)
    if user_update.municipal_access is not None:
        user.municipal_access = strip_suffix(user_update.municipal_access)
    
    db.commit()
    db.refresh(user)
    
    return {
        "message": "User access updated successfully",
        "user": {
            "id": user.id,
            "username": user.user_name,
            "provincial_access": user.provincial_access,
            "municipal_access": user.municipal_access
        },
        "updated_by": current_admin.user_name,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.put("/users/{user_id}")
async def update_user(
    user_id: int,
    user_data: dict,
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_auth_db)
):
    """Update all user details"""
    user = db.query(User).filter(User.id == user_id).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Update user fields
    if "first_name" in user_data:
        user.first_name = user_data["first_name"]
    if "last_name" in user_data:
        user.last_name = user_data["last_name"]
    if "email" in user_data:
        user.email = user_data["email"]
    if "contact_number" in user_data:
        user.contact_number = user_data["contact_number"]

    # ✅ Normalize PSA codes before saving
    if "provincial_access" in user_data:
        user.provincial_access = strip_suffix(user_data["provincial_access"])
    if "municipal_access" in user_data:
        user.municipal_access = strip_suffix(user_data["municipal_access"])
    
    try:
        db.commit()
        db.refresh(user)
        
        return {
            "message": "User updated successfully",
            "user": {
                "id": user.id,
                "username": user.user_name,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "email": user.email,
                "contact_number": user.contact_number,
                "provincial_access": user.provincial_access,
                "municipal_access": user.municipal_access
            },
            "updated_by": current_admin.user_name,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update user: {str(e)}")


@router.delete("/users/{user_id}")
async def delete_user(user_id: int,
                      current_admin: Admin = Depends(get_current_admin),
                      db: Session = Depends(get_auth_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    username = user.user_name
    db.delete(user)
    db.commit()

    return {
        "message": f"User '{username}' deleted successfully",
        "deleted_by": current_admin.user_name,
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/users/all")
async def get_all_users_and_requests(
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_auth_db)
):
    """Get all users and registration requests with their status"""
    
    all_requests = db.query(UserRegistrationRequest).order_by(
        UserRegistrationRequest.status,
        UserRegistrationRequest.request_date.desc()
    ).all()
    
    all_users = db.query(User).order_by(User.id.desc()).all()
    
    pending_requests = [r for r in all_requests if r.status == 'pending']
    approved_requests = [r for r in all_requests if r.status == 'approved']
    rejected_requests = [r for r in all_requests if r.status == 'rejected']
    
    return {
        "pending_requests": [
            {
                "id": req.id,
                "type": "registration_request",
                "username": req.user_name,
                "first_name": req.first_name,
                "last_name": req.last_name,
                "email": req.email,
                "contact_number": req.contact_number,
                "requested_provincial_access": req.requested_provincial_access,
                "requested_municipal_access": req.requested_municipal_access,
                "requested_provincial_code": req.requested_provincial_code,
                "requested_municipal_code": req.requested_municipal_code,
                "is_available": req.is_available,   # ✅ always shown
                "request_date": req.request_date.isoformat() if req.request_date else None,
                "status": "pending"
            } for req in pending_requests
        ],
        "approved_users": [
            {
                "id": user.id,
                "type": "user",
                "username": user.user_name,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "email": user.email,
                "contact_number": user.contact_number,
                "provincial_access": strip_suffix(user.provincial_access),
                "municipal_access": strip_suffix(user.municipal_access),
                "has_full_access": (
                    user.municipal_access and user.municipal_access.strip().lower() == "all"
                ),
                "access_description": (
                    "All municipalities under this province"
                    if user.municipal_access and user.municipal_access.strip().lower() == "all"
                    else strip_suffix(user.municipal_access)
                ),
                "status": "active"
            } for user in all_users
        ],

        "rejected_requests": [
            {
                "id": req.id,
                "username": req.user_name,
                "first_name": req.first_name,
                "last_name": req.last_name,
                "email": req.email,
                "rejected_date": req.review_date.isoformat() if req.review_date else None,
                "rejected_by": req.reviewed_by,
                "reason": req.remarks,
                "requested_provincial_code": req.requested_provincial_code,
                "requested_municipal_code": req.requested_municipal_code,
                "is_available": req.is_available,   # ✅ also shown here
                "status": "rejected"
            } for req in rejected_requests
        ],
        "statistics": {
            "total_pending": len(pending_requests),
            "total_approved": len(all_users),
            "total_rejected": len(rejected_requests),
            "total_all": len(all_requests) + len(all_users)
        }
    }


# Specific endpoint for pending only
@router.get("/users/pending")
async def get_pending_users(
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_auth_db)
):
    """Get only pending registration requests"""
    
    # Get pending registration requests
    pending_requests = db.query(UserRegistrationRequest).filter(
        UserRegistrationRequest.status == 'pending'
    ).order_by(UserRegistrationRequest.request_date.desc()).all()
    
    return {
        "pending_requests": [
            {
                "id": req.id,
                "username": req.user_name,
                "first_name": req.first_name,
                "last_name": req.last_name,
                "email": req.email,
                "contact_number": req.contact_number,
                "requested_provincial_access": req.requested_provincial_access,
                "requested_municipal_access": req.requested_municipal_access,
                "request_date": req.request_date.isoformat() if req.request_date else None,
                "days_pending": (datetime.utcnow() - req.request_date).days if req.request_date else 0
            } for req in pending_requests
        ],
        "total_pending": len(pending_requests)
    }

# Update the existing statistics endpoint to include registration info
@router.get("/statistics")
async def get_admin_statistics(
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_auth_db)
):
    """Get system statistics for admin dashboard"""
    try:
        # User statistics
        total_users = db.query(User).count()
        users_with_full_access = db.query(User).filter(
            User.provincial_access.isnot(None),
            User.municipal_access.isnot(None)
        ).count()
        users_with_partial_access = db.query(User).filter(
            or_(  # Use imported or_ instead of db.or_
                and_(User.provincial_access.isnot(None), User.municipal_access.is_(None)),
                and_(User.provincial_access.is_(None), User.municipal_access.isnot(None))
            )
        ).count()
        users_no_access = db.query(User).filter(
            User.provincial_access.is_(None),
            User.municipal_access.is_(None)
        ).count()
        
        # Registration request statistics
        pending_registrations = db.query(UserRegistrationRequest).filter(
            UserRegistrationRequest.status == 'pending'
        ).count()
        approved_registrations = db.query(UserRegistrationRequest).filter(
            UserRegistrationRequest.status == 'approved'
        ).count()
        rejected_registrations = db.query(UserRegistrationRequest).filter(
            UserRegistrationRequest.status == 'rejected'
        ).count()
        
        # Recent activity (last 7 days)
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        recent_registrations = db.query(UserRegistrationRequest).filter(
            UserRegistrationRequest.request_date >= seven_days_ago
        ).count()
        
        return {
            "users": {
                "total": total_users,
                "with_full_access": users_with_full_access,
                "with_partial_access": users_with_partial_access,
                "no_access": users_no_access
            },
            "registrations": {
                "pending": pending_registrations,
                "approved": approved_registrations,
                "rejected": rejected_registrations,
                "total": pending_registrations + approved_registrations + rejected_registrations,
                "recent_7_days": recent_registrations
            },
            "generated_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        print(f"Error in statistics endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating statistics: {str(e)}"
        )

@router.get("/locations")
async def get_available_locations(
    db: Session = Depends(get_auth_db)
):
    """Get provinces and their municipalities grouped by PSGC codes"""
    try:
        # Provinces
        provinces_query = db.execute(text("""
            SELECT DISTINCT province_code, province_name
            FROM credentials_users_schema.psa_table
            ORDER BY province_name;
        """))
        provinces = [{"code": row[0], "name": row[1]} for row in provinces_query.fetchall()]

        # Municipalities tied to provinces
        municipalities_query = db.execute(text("""
            SELECT province_code, municipal_code, municipal_name
            FROM credentials_users_schema.psa_table
            ORDER BY municipal_name;
        """))
        municipalities_by_province = {}
        for prov_code, muni_code, muni_name in municipalities_query.fetchall():
            if prov_code not in municipalities_by_province:
                municipalities_by_province[prov_code] = []
            municipalities_by_province[prov_code].append({
                "code": muni_code,
                "name": muni_name
            })

        return {
            "provinces": provinces,
            "municipalities": municipalities_by_province
        }

    except Exception as e:
        print(f"Error fetching PSGC locations: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching PSGC locations: {str(e)}"
        )



@router.get("/statistics")
async def get_admin_statistics(
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_auth_db)
):
    """Get system statistics for admin dashboard"""
    try:
        # User statistics
        total_users = db.query(User).count()
        users_with_full_access = db.query(User).filter(
            User.provincial_access.isnot(None),
            User.municipal_access.isnot(None)
        ).count()
        users_with_partial_access = db.query(User).filter(
            or_(
                and_(User.provincial_access.isnot(None), User.municipal_access.is_(None)),
                and_(User.provincial_access.is_(None), User.municipal_access.isnot(None))
            )
        ).count()
        users_no_access = db.query(User).filter(
            User.provincial_access.is_(None),
            User.municipal_access.is_(None)
        ).count()
        
        # Registration request statistics
        pending_registrations = db.query(UserRegistrationRequest).filter(
            UserRegistrationRequest.status == 'pending'
        ).count()
        approved_registrations = db.query(UserRegistrationRequest).filter(
            UserRegistrationRequest.status == 'approved'
        ).count()
        rejected_registrations = db.query(UserRegistrationRequest).filter(
            UserRegistrationRequest.status == 'rejected'
        ).count()
        
        # Recent activity (last 7 days)
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        recent_registrations = db.query(UserRegistrationRequest).filter(
            UserRegistrationRequest.request_date >= seven_days_ago
        ).count()
        
        return {
            "users": {
                "total": total_users,
                "with_full_access": users_with_full_access,
                "with_partial_access": users_with_partial_access,
                "no_access": users_no_access
            },
            "registrations": {
                "pending": pending_registrations,
                "approved": approved_registrations,
                "rejected": rejected_registrations,
                "total": pending_registrations + approved_registrations + rejected_registrations,
                "recent_7_days": recent_registrations
            },
            "generated_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        print(f"Error in statistics endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating statistics: {str(e)}"
        )

@router.get("/registration-requests")
async def get_registration_requests(status: str = "pending",
                                    current_admin: Admin = Depends(get_current_admin),
                                    db: Session = Depends(get_auth_db)):
    query = db.query(UserRegistrationRequest)
    if status != 'all':
        query = query.filter(UserRegistrationRequest.status == status)

    requests = query.order_by(UserRegistrationRequest.request_date.desc()).all()
    return {
        "requests": [
            {
                "id": r.id,
                "username": r.user_name,
                "first_name": r.first_name,
                "last_name": r.last_name,
                "email": r.email,
                "contact_number": r.contact_number,
                "requested_provincial_access": strip_suffix(r.requested_provincial_access),
                "requested_municipal_access": strip_suffix(r.requested_municipal_access),
                "requested_provincial_code": strip_suffix(r.requested_provincial_code),
                "requested_municipal_code": strip_suffix(r.requested_municipal_code),
                "is_available": r.is_available,
                "status": r.status,
                "request_date": r.request_date.isoformat() if r.request_date else None,
                "reviewed_by": r.reviewed_by,
                "review_date": r.review_date.isoformat() if r.review_date else None,
                "remarks": r.remarks
            }
            for r in requests
        ],
        "total": len(requests)
    }


@router.post("/review-registration")
async def review_registration(
    review: RegistrationReviewRequest,
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_auth_db)
):
    """Approve or reject a registration request"""
    try:
        # 🔎 Get the registration request
        reg_request = db.query(UserRegistrationRequest).filter(
            UserRegistrationRequest.id == review.request_id
        ).first()
        
        if not reg_request:
            raise HTTPException(status_code=404, detail="Registration request not found")
        
        if reg_request.status != 'pending':
            raise HTTPException(
                status_code=400,
                detail=f"Request already {reg_request.status}"
            )
        
        # ✏️ Update metadata for the review
        reg_request.reviewed_by = current_admin.user_name
        reg_request.review_date = datetime.utcnow()
        reg_request.remarks = review.remarks
        
        if review.action == 'approve':
            # 🚫 Cannot approve if LGU not available
            if not reg_request.is_available:
                raise HTTPException(
                    status_code=400,
                    detail="Cannot approve request: selected province/municipality not available"
                )
            
            # ✅ Check duplicates
            existing_user = db.query(User).filter(
                (User.user_name == reg_request.user_name) | 
                (User.email == reg_request.email)
            ).first()
            if existing_user:
                raise HTTPException(
                    status_code=409,
                    detail="Username or email already exists"
                )
            
            # 🆕 Create new user with stripped PSA codes
            new_user = User(
                user_name=reg_request.user_name,
                password=reg_request.password,  # already hashed
                first_name=reg_request.first_name,
                last_name=reg_request.last_name,
                email=reg_request.email,
                contact_number=reg_request.contact_number,
                provincial_access=strip_suffix(review.provincial_access or reg_request.requested_provincial_access),
                municipal_access=strip_suffix(review.municipal_access or reg_request.requested_municipal_access)
            )
            
            db.add(new_user)
            reg_request.status = 'approved'
            db.commit()
            db.refresh(new_user)
            
            return {
                "message": f"Registration approved for user {reg_request.user_name}",
                "user_id": new_user.id,
                "provincial_access": new_user.provincial_access,
                "municipal_access": new_user.municipal_access,
                "status": "success",
                "approved_by": current_admin.user_name,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        elif review.action == 'reject':
            reg_request.status = 'rejected'
            db.commit()
            
            return {
                "message": f"Registration rejected for user {reg_request.user_name}",
                "status": "success",
                "rejected_by": current_admin.user_name,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        else:
            raise HTTPException(status_code=400, detail="Invalid action. Use 'approve' or 'reject'")
    
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error processing registration: {str(e)}")



@router.put("/registration-requests/{request_id}")
async def update_registration_request(
    request_id: int,
    request_data: dict,
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_auth_db)
):
    """Update registration request details"""
    reg_request = db.query(UserRegistrationRequest).filter(
        UserRegistrationRequest.id == request_id
    ).first()
    
    if not reg_request:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Registration request not found"
        )
    
    # Update request fields
    if "first_name" in request_data:
        reg_request.first_name = request_data["first_name"]
    if "last_name" in request_data:
        reg_request.last_name = request_data["last_name"]
    if "email" in request_data:
        reg_request.email = request_data["email"]
    if "contact_number" in request_data:
        reg_request.contact_number = request_data["contact_number"]
    if "requested_provincial_access" in request_data:
        reg_request.requested_provincial_access = request_data["requested_provincial_access"]
    if "requested_municipal_access" in request_data:
        reg_request.requested_municipal_access = request_data["requested_municipal_access"]
    
    try:
        db.commit()
        db.refresh(reg_request)
        
        return {
            "message": "Registration request updated successfully",
            "request": {
                "id": reg_request.id,
                "username": reg_request.user_name,
                "first_name": reg_request.first_name,
                "last_name": reg_request.last_name,
                "email": reg_request.email,
                "contact_number": reg_request.contact_number,
                "requested_provincial_access": reg_request.requested_provincial_access,
                "requested_municipal_access": reg_request.requested_municipal_access,
                "status": reg_request.status
            },
            "updated_by": current_admin.user_name,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update registration request: {str(e)}"
        )

# Registration requests statistics
@router.get("/registration-statistics")
async def get_registration_statistics(
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_auth_db)
):
    """Get statistics about registration requests"""
    pending_count = db.query(UserRegistrationRequest).filter(
        UserRegistrationRequest.status == 'pending'
    ).count()
    
    approved_count = db.query(UserRegistrationRequest).filter(
        UserRegistrationRequest.status == 'approved'
    ).count()
    
    rejected_count = db.query(UserRegistrationRequest).filter(
        UserRegistrationRequest.status == 'rejected'
    ).count()
    
    # Get recent requests (last 7 days)
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    recent_requests = db.query(UserRegistrationRequest).filter(
        UserRegistrationRequest.request_date >= seven_days_ago
    ).count()
    
    return {
        "statistics": {
            "pending_requests": pending_count,
            "approved_requests": approved_count,
            "rejected_requests": rejected_count,
            "total_requests": pending_count + approved_count + rejected_count,
            "recent_requests_7days": recent_requests
        },
        "generated_at": datetime.utcnow().isoformat()
    }

# Optional: Cleanup old rejected requests
@router.delete("/cleanup-rejected-requests")
async def cleanup_rejected_requests(
    days_old: int = 30,  # Delete rejected requests older than X days
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_auth_db)
):
    """Delete rejected requests older than specified days"""
    cutoff_date = datetime.utcnow() - timedelta(days=days_old)
    
    # First, get count of requests to be deleted
    requests_to_delete = db.query(UserRegistrationRequest).filter(
        UserRegistrationRequest.status == 'rejected',
        UserRegistrationRequest.review_date < cutoff_date
    ).all()
    
    deleted_usernames = [req.user_name for req in requests_to_delete]
    deleted_count = len(requests_to_delete)
    
    # Delete the requests
    db.query(UserRegistrationRequest).filter(
        UserRegistrationRequest.status == 'rejected',
        UserRegistrationRequest.review_date < cutoff_date
    ).delete()
    
    db.commit()
    
    return {
        "message": f"Deleted {deleted_count} rejected requests older than {days_old} days",
        "deleted_count": deleted_count,
        "deleted_usernames": deleted_usernames,
        "performed_by": current_admin.user_name,
        "timestamp": datetime.utcnow().isoformat()
    }

# Update the existing statistics endpoint to include registration requests
@router.get("/statistics")
async def get_admin_statistics(
    current_admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_auth_db)
):
    """Get system statistics for admin dashboard"""
    total_users = db.query(User).count()
    users_with_full_access = db.query(User).filter(
        User.provincial_access.isnot(None),
        User.municipal_access.isnot(None)
    ).count()
    users_pending_approval = db.query(User).filter(
        (User.provincial_access.is_(None)) | 
        (User.municipal_access.is_(None))
    ).count()
    
    # Add registration request statistics
    pending_registrations = db.query(UserRegistrationRequest).filter(
        UserRegistrationRequest.status == 'pending'
    ).count()
    
    return {
        "statistics": {
            "total_users": total_users,
            "users_with_full_access": users_with_full_access,
            "users_pending_approval": users_pending_approval,
            "users_with_partial_access": total_users - users_with_full_access - users_pending_approval,
            "pending_registrations": pending_registrations  # NEW
        },
        "generated_at": datetime.utcnow().isoformat()
    }